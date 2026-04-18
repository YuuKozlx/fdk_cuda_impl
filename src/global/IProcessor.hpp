#pragma once
#include <driver_types.h>
#include "tl/expected.hpp"

namespace YK {
    // ============================================================
    // ������/�˲��ӿ�
    // ============================================================
    class IProcessor {
    public:
        virtual ~IProcessor() = default;

        // ��ʼ���ӿڣ��̳е���ɸ�����Ҫ���Ӳ�����init() ֻ�����ʼ����Դ/plan��������Ȩ�أ�lazy��
        virtual bool init() = 0;

        // �����ӿڣ����������Ϊ device pointer��in-place ���
        virtual void process(const void* d_input, void* d_output, cudaStream_t stream = 0) = 0;

        // ��Դ�ͷŽӿڣ�����ʱ�Զ����ã�Ҳ���ֶ����� release() ����ǰ�ͷ���Դ
        virtual void release() = 0;

        // ״̬��ѯ�ӿڣ�������Ҫ���ӣ����Ƿ��ѳ�ʼ������ǰ���õ�
        virtual bool isInitialized() const = 0;

        // ���ƽӿڣ����ش��������ƣ�������־/����
        virtual const char* name() const = 0;

        // ������������ע�룬Ĭ�Ͽ�ʵ��
        // ��Ҫ per-chunk ״̬�����า�Ǵ˽ӿ�
        // ���÷��� process() ǰ����
        virtual void setContext(const void* /*ctx*/) {}

        // ��ʼ������ע�루init() ֮ǰ���ã�
        virtual void setInitContext(const void* /*ctx*/) {}

    };
} // namespace YK